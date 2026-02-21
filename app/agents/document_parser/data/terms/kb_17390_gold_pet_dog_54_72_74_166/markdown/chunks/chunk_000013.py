from langchain_core.documents import Document

chunk = Document(
    page_content=('10의2. 「공직선거법」제34조에 따른 임기만료에 의한 선거의 선거일\n'
 '11. 기타 정부에서 수시 지정하는 날# 제3조(대체공휴일)① 제2조제2호부터 제10호까지의 공휴일이 다음 각 호의 어느 하나에 해당하\n'
 '는 경우에는 그 공휴일 다음의 첫 번째 비공휴일(제2조 각 호의 공휴일이\n'
 '아닌 날을 말한다. 이하 같다)을 대체공휴일로 한다.- 1. 제2조제2호 또는 제7호의 공휴일이 토요일이나 일요일과 겹치는 경우\n'
 '- 별\n'
 '- 2. 제2조제4호 또는 제9호의 공휴일이 일요일과 겹치는 경우\n'
 '- 표'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
