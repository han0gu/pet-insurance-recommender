from langchain_core.documents import Document

chunk = Document(
    page_content=('이 특별약관에 있어서 "2대호흡계특정질환"이라</h1><br><p id=\'163\' data-category=\'paragraph\' '
 "style='font-size:14px'>함은 제9차 한국표준질병․사인분</p><br><p id='164' "
 "data-category='paragraph' style='font-size:14px'>류에 있어서 【별표11】(2대호흡계특정질환 "
 '분류표)에서 정한 질병을 말합니다.<br>\uf000 "2대호흡계특정질환"의 진단확정은 의료법 제3조(의료기관)에서 정한 국내의 '
 '병<br>원이나 의원 또는 국외의 의료관련법에서'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000634',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
