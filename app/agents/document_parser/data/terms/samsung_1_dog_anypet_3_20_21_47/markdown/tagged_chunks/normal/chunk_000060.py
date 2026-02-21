from langchain_core.documents import Document

chunk = Document(
    page_content=('- 수신을 확인하기 전까지는 그 전자문서는 송신되지 않은 것으로 봅니다. 회사는 전자문서가 수신\n'
 '- 되지 않은 것으로 확인되는 경우에는 제1항의 납입최고(독촉)기간을 설정하여 제1항에서 정한 내\n'
 '- 용을 서면(등기우편 등) 또는 전화(음성녹음)로 다시 알려 드립니다.\n'
 '- ④ 회사가 제1항에 따른 납입최고(독촉) 등을 전화(음성녹음)로 안내하고자 할 때 다음 각 호의 요건\n'
 '- 을 모두 충족하는 경우에 「보험업감독규정」 제4-36조 제3항에 따른 전자적 상품설명장치를 활\n'
 '- 용할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000060',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
