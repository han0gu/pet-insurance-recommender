from langchain_core.documents import Document

chunk = Document(
    page_content=('. 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변<br>4. 핵연료물질 또는 핵연료물질에 의하여 오염된 물질의 방사성, 폭발성 '
 "또는 그<br>밖의 유해한 특성에 의한 사고<br>부 가 설 명</p><br><table id='108' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>∙ 핵연료물질 : "
 '사용된</td><td>연료를 포함합니다.</td></tr><tr><td colspan="2">∙ 핵연료물질에 의하여 오염된 물질 : '
 '원자핵 분열 생성물을 포함합니다'),
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
 'indexing': {'chunk_id': 'chunk_001113',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
