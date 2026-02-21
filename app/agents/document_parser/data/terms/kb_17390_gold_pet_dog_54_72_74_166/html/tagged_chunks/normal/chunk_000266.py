from langchain_core.documents import Document

chunk = Document(
    page_content=('관한 법률" 제47조 및 관련규정이 정하는 바에 따라</p><br><p id=\'74\' data-category=\'list\' '
 "style='font-size:14px'>계약체결에 대한 회사의 법위반사항이 있는 경우 계약체결일부터 5년 이내의 범위<br>에서 "
 '계약자가 위반사항을 안 날부터 1년 이내에 계약해지요구서에 증빙서류를 첨<br>부하여 위법계약의 해지를 요구할 수 '
 '있습니다.<br>\uf000 회사는 해지요구를 받은 날부터 10일 이내에 수락여부를 계약자에 통지하여야 하<br>며, 거절할 때에는 거절 '
 '사유를 함께 통지하여야'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000266',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
