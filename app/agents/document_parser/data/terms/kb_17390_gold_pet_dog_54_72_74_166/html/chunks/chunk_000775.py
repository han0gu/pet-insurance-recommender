from langchain_core.documents import Document

chunk = Document(
    page_content=(". 기타 정부에서 수시 지정하는 날</p><br><p id='131' data-category='paragraph' "
 "style='font-size:16px'>제3조(대체공휴일)</p><br><p id='132' "
 "data-category='paragraph' style='font-size:14px'>① 제2조제2호부터 제10호까지의 공휴일이 다음 "
 '각 호의 어느 하나에 해당하<br>는 경우에는 그 공휴일 다음의 첫 번째 비공휴일(제2조 각 호의 공휴일이<br>아닌 날을 말한다'),
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
