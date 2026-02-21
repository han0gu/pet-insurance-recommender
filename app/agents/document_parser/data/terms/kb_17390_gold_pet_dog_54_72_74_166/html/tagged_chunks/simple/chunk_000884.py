from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사는<br>전자문서가 수신되지 않은 것을 확인한 경우에는 제1항에서 정한 내용을 서면(등<br>기우편 등) 또는 전화(음성녹음)로 '
 '다시 알려 드립니다.<br>\uf000 회사가 제1항에 따른 납입최고(독촉) 등을 전화(음성녹음)로 안내하고자 할 때<br>다음 각 호의 '
 '요건을 모두 충족하는 경우에 보험업감독규정 제4-36조 제3항에 따<br>른 전자적 상품설명장치를 활용할 수 있습니다.<br>1'),
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
 'indexing': {'chunk_id': 'chunk_000884',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
