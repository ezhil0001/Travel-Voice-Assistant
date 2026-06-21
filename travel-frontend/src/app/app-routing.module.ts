import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

// The chat feature is lazy-loaded so the initial bundle stays small.
// Any future feature (settings, history, etc.) can follow the same pattern.
const routes: Routes = [
  { path: '', redirectTo: '/chat', pathMatch: 'full' },
  {
    path: 'chat',
    loadChildren: () =>
      import('./features/chat/chat.module').then(m => m.ChatModule)
  }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
