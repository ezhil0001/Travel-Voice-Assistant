import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterModule, Routes } from '@angular/router';

import { SharedModule } from '../../shared/shared.module';
import { ChatComponent } from './chat.component';
import { ChatWindowComponent } from './components/chat-window/chat-window.component';
import { MessageBubbleComponent } from './components/message-bubble/message-bubble.component';
import { VoicePanelComponent } from './components/voice-panel/voice-panel.component';

const routes: Routes = [
  { path: '', component: ChatComponent },
];

@NgModule({
  declarations: [
    ChatComponent,
    ChatWindowComponent,
    MessageBubbleComponent,
    VoicePanelComponent,
  ],
  imports: [
    CommonModule,
    FormsModule,
    SharedModule,
    RouterModule.forChild(routes),
  ],
})
export class ChatModule {}
