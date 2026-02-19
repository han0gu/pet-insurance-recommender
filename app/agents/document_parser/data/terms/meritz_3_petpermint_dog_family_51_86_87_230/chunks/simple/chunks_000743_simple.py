from langchain_core.documents import Document

chunk = Document(
    page_content=("가) 척추체(척추뼈 몸통)의 만곡변화는 객관적인 측정방 법(Cobb's Angle)에 따라 골절이 발생한 척추체(척 추뼈 몸통)의 상ㆍ하 "
 '인접 정상 척추체(척추뼈 몸 통)를 포함하여 측정하며, 생리적 정상만곡을 고려 하여 평가한다. 나) 척추(등뼈)의 기형장해는 '
 '척추체(척추뼈 몸통)의 압 박률, 골절의 부위 등을 기준으로 판정한다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 211},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000743',
              'chunk_char_len': 185,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
