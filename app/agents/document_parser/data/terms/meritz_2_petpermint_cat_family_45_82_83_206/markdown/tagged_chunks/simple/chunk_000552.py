from langchain_core.documents import Document

chunk = Document(
    page_content=('다)의 압박률 또는 척추체(척추뼈 몸통)의 만곡 정도에\n'
 '따라 평가한다.- 가) 척추체(척추뼈 몸통)의 만곡변화는 객관적인 측정방\n'
 "- 법(Cobb's Angle)에 따라 골절이 발생한 척추체(척\n"
 '- 추뼈 몸통)의 상ㆍ하 인접 정상 척추체(척추뼈 몸\n'
 '- 통)를 포함하여 측정하며, 생리적 정상만곡을 고려\n'
 '- 하여 평가한다.\n'
 '- 나) 척추(등뼈)의 기형장해는 척추체(척추뼈 몸통)의 압\n'
 '- 박률, 골절의 부위 등을 기준으로 판정한다. 척추체\n'
 '- (척추뼈 몸통)의 압박률은 인접 상ㆍ하부[인접 상ㆍ'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000552',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
