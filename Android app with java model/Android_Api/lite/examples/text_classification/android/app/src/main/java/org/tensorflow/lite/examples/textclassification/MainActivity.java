package org.tensorflow.lite.examples.textclassification;/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//package org.tensorflow.lite.examples.textclassification.GFG;
import org.tensorflow.lite.examples.textclassification.logistic;
import java.util.Dictionary;
import java.util.Hashtable;
import java.util.List;
import android.content.Intent;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.support.annotation.WorkerThread;
import android.util.Log;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
//import java.util.ArrayList;
import android.widget.Button;
//import java.util.ArrayList;
//import java.util.HashMap;
//import java.util.List;
//import java.util.Map;
//import java.util.PriorityQueue;
//import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.textclassification.R;

import android.os.Bundle;
import android.os.Handler;
import android.speech.RecognizerIntent;
import android.support.v7.app.AppCompatActivity;
import android.widget.EditText;
import android.widget.ScrollView;
import android.widget.TextView;
import android.view.View;
import android.widget.Toast;


import java.util.Locale;
import java.util.ArrayList;
/** The main activity to provide interactions with users. */
public class MainActivity extends AppCompatActivity {
  private static final String TAG = "TextClassificationDemo";
//  private TextClassificationClient client;
//  private TextView txvResult;
  private TextView resultTextView;
  private EditText inputEditText;
  private Handler handler;
  private ScrollView scrollView;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.tfe_tc_activity_main);
//    txvResult = (TextView) findViewById(R.id.txvResult);
//    Log.v(TAG, "onCreate");

//    client = new TextClassificationClient(getApplicationContext());
//    handler = new Handler();
      resultTextView = findViewById(R.id.result_text_view);
      inputEditText = findViewById(R.id.input_text);
      scrollView = findViewById(R.id.scroll_view);
    Button classifyButton = findViewById(R.id.button);
    classifyButton.setOnClickListener(
        (View v) -> {
          resultTextView.setText(org.tensorflow.lite.examples.textclassification.logistic.classifyy(inputEditText.getText().toString()));
        });

  }

//  @Override
//  protected void onStart() {
//    super.onStart();
//    Log.v(TAG, "onStart");
//    handler.post(
//        () -> {
//          client.load();
//        });
//  }

//  @Override
//  protected void onStop() {
//    super.onStop();
//    Log.v(TAG, "onStop");
//    handler.post(
//        () -> {
//          client.unload();
//        });
//  }
//  @Override











  public void getSpeechInput(View view) {

      Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
      intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
      intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault());

      if (intent.resolveActivity(getPackageManager()) != null) {
          startActivityForResult(intent, 10);
      } else {
          Toast.makeText(this, "Your Device Don't Support Speech Input", Toast.LENGTH_SHORT).show();
      }
  }
  @Override
  protected void onActivityResult(int requestCode, int resultCode, Intent data) {
      super.onActivityResult(requestCode, resultCode, data);

      switch (requestCode) {
          case 10:
              if (resultCode == RESULT_OK && data != null) {
                  ArrayList<String> result = data.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS);
//                  txvResult.setText(result.get(0));
                  inputEditText.setText(result.get(0));
              }
              break;
      }
  }

  // Send input text to TextClassificationClient and get the classify messages.
//  private void classify(final String text) {
//    handler.post(
//        () -> {
//          // Run text classification with TF Lite.
//          List<TextClassificationClient.Result> results = client.classify(text);
//
//          // Show classification result on screen
//          showResult(text, results);
//        });
//  }

  //Show classification result on the screen.
//  private void showResult(final String inputText, final List<TextClassificationClient.Result> results) {
//      Dictionary dict = new Hashtable();
//
//      // add elements in the hashtable
//      dict.put("1", "Undo");
//      dict.put("2", "Bold");
//      dict.put("3", "Unbold");
//      dict.put("4", "italcs");
//      dict.put("5", "remove italics");
//      dict.put("6", "underline");
//      dict.put("7", "remove underline");
//      dict.put("8", "superscript");
//      dict.put("9", "remove superscript");
//      dict.put("10", "subscript");
//      dict.put("11", "remove subscript");
//      dict.put("12", "strike");
//      dict.put("13", "remove strike");
//      dict.put("14", "centre align");
//      dict.put("15", "insert comment");
//      dict.put("16", "left align");
//      dict.put("17", "right align");
//      dict.put("18", "remove formating");
//      dict.put("19", "insert bullet");
//      dict.put("20", "next bullet");
//      dict.put("21", "stop bullets");
//      dict.put("22", "pause dict");
//      dict.put("23", "stop dict");
//      dict.put("24", "show commands");
//      dict.put("25", "help");
//      dict.put("26", "delete");
//
//
//      // Run on UI thread as we'll updating our app UI
//    runOnUiThread(
//        () -> {
//          String textToShow = "Input: " + inputText + "\nOutput:\n";
//          for (int i = 0; i < results.size(); i++) {
//            TextClassificationClient.Result result = results.get(i);
//            textToShow +=
//                String.format("    %s: %s\n", dict.get(result.getTitle()), result.getConfidence());
//          }
//          textToShow += "---------\n";
//
//          // Append the result to the UI.
//          resultTextView.append(textToShow);
//
//          // Clear the input text.
//          inputEditText.getText().clear();
//
//          // Scroll to the bottom to show latest entry's classification result.
//          scrollView.post(() -> scrollView.fullScroll(View.FOCUS_DOWN));
//        });
//  }
}
