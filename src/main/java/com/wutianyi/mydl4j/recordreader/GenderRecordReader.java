package com.wutianyi.mydl4j.recordreader;

import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

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
			}
		}
	}
}
