import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { MicButtonComponent } from './components/mic-button/mic-button.component';
import { AudioWaveComponent } from './components/audio-wave/audio-wave.component';
import { ToolActivityComponent } from './components/tool-activity/tool-activity.component';
import { ToastComponent } from './components/toast/toast.component';
import { MarkdownPipe } from './pipes/markdown.pipe';

@NgModule({
  declarations: [
    MicButtonComponent,
    AudioWaveComponent,
    ToolActivityComponent,
    ToastComponent,
    MarkdownPipe,
  ],
  imports: [CommonModule],
  exports: [
    CommonModule,
    MicButtonComponent,
    AudioWaveComponent,
    ToolActivityComponent,
    ToastComponent,
    MarkdownPipe,
  ],
})
export class SharedModule {}

