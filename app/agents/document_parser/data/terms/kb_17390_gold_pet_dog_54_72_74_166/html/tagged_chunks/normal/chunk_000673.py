from langchain_core.documents import Document

chunk = Document(
    page_content=('style=\'font-size:14px\'>\uf000 이 특별약관에 있어서 "호흡기관련질병"이라 함은【별표14】(호흡기관련질병 '
 '분<br>류표)에서 정한 질병을 말합니다.<br>\uf000 "호흡기관련질병"의 진단확정은 의료법 제3조에서 정한 국내의 병원이나 의원 '
 '또<br>는 국외의 의료관련법에서 정한 의료기관의 의사자격을 가진 자에 의한 진단서에<br>의합니다'),
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
 'indexing': {'chunk_id': 'chunk_000673',
              'chunk_char_len': 192,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
