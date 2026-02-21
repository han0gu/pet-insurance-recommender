from langchain_core.documents import Document

chunk = Document(
    page_content=("id='136' data-category='list' style='font-size:14px'>제2조제2호 또는 제7호의 공휴일이 "
 '토요일이나 일요일과 겹치는 경우<br>해<br>제2조제4호 또는 제9호의 공휴일이 일요일과 겹치는 경우 '
 '및<br>제2조제2호ㆍ제4호ㆍ제7호 또는 제9호의 공휴일이 토요일ㆍ일요일이 질<br>아닌 날에 같은 조 제2호부터 제10호까지의 규정에 '
 "따른 다른 공휴일 병<br>과 겹치는 경우</p><br><p id='137' data-category='paragraph'"),
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
