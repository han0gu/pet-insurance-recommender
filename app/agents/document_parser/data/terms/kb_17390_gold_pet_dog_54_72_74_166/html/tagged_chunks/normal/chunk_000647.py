from langchain_core.documents import Document

chunk = Document(
    page_content=('"6대호흡계특정질환"이라 함은 제9차 한국표준질병․사인분<br>특<br>류에 있어서 【별표12】(6대호흡계특정질환 분류표)에서 정한 '
 '질병을 말합니다.<br>별<br>\uf000 "6대호흡계특정질환"의 진단확정은 의료법 제3조(의료기관)에서 정한 국내의 '
 '병<br>약<br>원이나 의원 또는 국외의 의료관련법에서 정한 의료기관의 의사(치과의사 제외)<br>관<br>면허를 가진 자에 의하여 '
 '내려져야 합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000647',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
