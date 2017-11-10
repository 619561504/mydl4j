package com.wutianyi.mydl4j.recordreader;

import org.apache.commons.lang.StringUtils;
import org.datavec.api.berkeley.Pair;
import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class GenderRecordReader extends LineRecordReader {

	private List<String> labels;

	private List<String> names = new ArrayList<>();

	private String possibleCharacters = "";

	public int maxLengthName = 0;

	private int totalRecords = 0;

	private Iterator<String> iter;

	public GenderRecordReader(List<String> labels) {
		this.labels = labels;
	}

	private int totalRecords() {
		return totalRecords;
	}

	@Override
	public void initialize(InputSplit split) throws IOException, InterruptedException {

		if (split instanceof FileSplit) {
			URI[] locations = split.locations();
			if (locations != null && locations.length > 1) {
				String longestName = "";
				String uniqueCharactersTempString = "";
				List<Pair<String, List<String>>> tempNames = new ArrayList<>();
				for (URI location : locations) {
					File file = new File(location);
					List<String> temp = this.labels.stream().filter(line -> file.getName().equals(line + ".csv"))
							.collect(Collectors.toList());
					if (temp.size() > 0) {
						Path path = Paths.get(file.getAbsolutePath());
						List<String> tempList = Files.readAllLines(path, StandardCharsets.UTF_8).stream()
								.map(element -> element.split(",")[0]).collect(Collectors.toList());

						Optional<String> optional = tempList.stream().reduce((name1, name2) -> name1.length() > name2.length() ? name1 : name2);
						if (optional.isPresent() && optional.get().length() > longestName.length()) {
							longestName = optional.get();
						}
						uniqueCharactersTempString = uniqueCharactersTempString + tempList.toString();
						Pair<String, List<String>> tempPair = new Pair<>(temp.get(0), tempList);
						tempNames.add(tempPair);
					} else
						throw new InterruptedException("File miss for any of the specified labels");
				}

				this.maxLengthName = longestName.length();
				String unique = Stream.of(uniqueCharactersTempString).map(w -> w.split("")).flatMap(Arrays::stream)
						.distinct().collect(Collectors.toList()).toString();
				char[] chars = unique.toCharArray();
				Arrays.sort(chars);
				unique = new String(chars);
				unique = unique.replaceAll("\\[", "").replaceAll("\\]", "").replaceAll(",", "");
				if (unique.startsWith(" ")) {
					unique = " " + unique.trim();
				}
				this.possibleCharacters = unique;
				System.out.println(unique);
				Pair<String, List<String>> tempPair = tempNames.get(0);
				int minSize = tempPair.getSecond().size();
				for (int i = 1; i < tempNames.size(); i++) {
					if (minSize > tempNames.get(i).getSecond().size())
						minSize = tempNames.get(i).getSecond().size();
				}

				List<Pair<String, List<String>>> oneMoreTempNames = new ArrayList<>();
				for (int i = 0; i < tempNames.size(); i++) {
					int diff = Math.abs(minSize - tempNames.get(i).getSecond().size());
					List<String> tempList = new ArrayList<>();

					if (tempNames.get(i).getSecond().size() > minSize) {
						tempList = tempNames.get(i).getSecond();
						tempList = tempList.subList(0, tempList.size() - diff);
					} else
						tempList = tempNames.get(i).getSecond();
					Pair<String, List<String>> tempNewPair = new Pair<>(tempNames.get(i).getFirst(), tempList);
					oneMoreTempNames.add(tempNewPair);
				}
				tempNames.clear();

				List<Pair<String, List<String>>> secondMoreTempNames = new ArrayList<>();
				for (int i = 0; i < oneMoreTempNames.size(); i++) {
					int gender = oneMoreTempNames.get(i).getFirst().equals("M") ? 1 : 0;
					List<String> secondList = oneMoreTempNames.get(i).getSecond().stream().map(element -> getBinaryString(element.split(",")[0], gender)).collect(Collectors.toList());
					Pair<String, List<String>> secondTempPair = new Pair<>(oneMoreTempNames.get(i).getFirst(), secondList);
					secondMoreTempNames.add(secondTempPair);
				}
				oneMoreTempNames.clear();

				for (int i = 0; i < secondMoreTempNames.size(); i++) {
					names.addAll(secondMoreTempNames.get(i).getSecond());
				}
				secondMoreTempNames.clear();
				this.totalRecords = names.size();
				Collections.shuffle(names);
				this.iter = names.iterator();
			}
		}
	}

	@Override
	public List<Writable> next() {
		if (iter.hasNext()) {
			List<Writable> ret = new ArrayList<>();
			String currentRecord = iter.next();
			String[] temp = currentRecord.split(",");
			for (int i = 0; i < temp.length; i++) {
				ret.add(new DoubleWritable(Double.parseDouble(temp[i])));
			}
			return ret;
		} else {
			throw new IllegalStateException("no more elements");
		}
	}

	@Override
	public boolean hasNext() {
		if (iter != null) {
			return iter.hasNext();
		}
		throw new IllegalStateException("Indeterminant state: record must not be null, or a file" +
				"iterator must exist");
	}

	private String getBinaryString(String name, int gender) {
		String binaryString = "";
		for (int j = 0; j < name.length(); j++) {
			String fs = StringUtils.leftPad(Integer.toBinaryString(this.possibleCharacters.indexOf(name.charAt(j))), 5, "0");
			binaryString = binaryString + fs;
		}

		binaryString = StringUtils.rightPad(binaryString, this.maxLengthName * 5, "0");
		binaryString = binaryString.replaceAll(".(?!$)", "$0,");
		return binaryString + "," + String.valueOf(gender);
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		GenderRecordReader genderRecordReader = new GenderRecordReader(new ArrayList<String>() {{
			add("M");
			add("F");
		}});

		String filePath = System.getProperty("user.dir") + "\\src\\main\\resources\\PredictGender\\Data\\";

		FileSplit fileSplit = new FileSplit(new File(filePath));
		genderRecordReader.initialize(fileSplit);
		System.out.println(genderRecordReader.next());
	}

	@Override
	public void reset() {
		this.iter = names.iterator();
	}
}
