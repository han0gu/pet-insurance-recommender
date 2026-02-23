from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해판정기준</p><br><p id='184' data-category='list' style='font-size:16px'>1) "
 '청력장해는 순음청력검사 결과에 따라 데시벨(dB: decibel)로서 표시하<br>고 3회 이상 청력검사를 실시한 후 적용한다'),
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
