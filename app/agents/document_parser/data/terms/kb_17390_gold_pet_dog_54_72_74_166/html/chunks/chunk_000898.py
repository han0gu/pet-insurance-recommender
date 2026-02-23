from langchain_core.documents import Document

chunk = Document(
    page_content=('. 해<br>\uf000 회사는 제1항의 통지를 계약이 해지된 날부터 7일 이내에 하여야 합니다.<br>\uf000 보험수익자는 통지를 '
 '받은 날(제3항에 따라 계약자에게 통지된 경우에는 계약자<br>가 통지를 받은 날을 말합니다)부터 15일 이내에 제1항의 절차를 이행할 '
 "수 있습</p><br><p id='68' data-category='paragraph' "
 "style='font-size:16px'>니다.</p><br><table id='69' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>용 어 풀"),
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
