from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 지급사유 관련 용어\n'
 '용어 | 정의\n'
 '상해 | 보험기간 중에 발생한 급격하고도 우연한 외 래의 사고로 신체(의수, 의족, 의안, 의치 등 신체보조장구는 제외하나, 인공장기나 '
 '부분 의치 등 신체에 이식되어 그 기능을 대신할 경우는 포함합니다)에 입은 상해를 말합니다.\n'
 '신체 | 의수, 의족, 의안, 의치 등 신체보조장구는 제외하나, 인공장기나 부분 의치 등 신체에 이식되어 그 기능을 대신할 경우는 '
 '포함합니 다.\n'
 '장해 | 【별표2(장해분류표)】에서 정한 기준에 따른 장해상태를 말합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 47},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000002',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
