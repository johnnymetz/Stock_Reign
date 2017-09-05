function getLastWeekday(date){
	var dayOfWeek = date.getUTCDay();  // Sunday = 0; Saturday = 6
	if (dayOfWeek == 6){
		date.setDate(date.getUTCDate() - 1);
	} else if (dayOfWeek == 0){
		date.setDate(date.getUTCDate() - 2);
	}
	date_str = (date.getUTCMonth() + 1) + '/' + date.getUTCDate() + '/' + date.getUTCFullYear()
	return date_str
}

function adjustDate(newStart, button_selector){
	$('#start_date').datepicker('setDate', getLastWeekday(newStart));
  $('.justify-buttons button').removeClass('active');
  button_selector.addClass('active');
}
