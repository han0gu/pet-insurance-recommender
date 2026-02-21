from langchain_core.documents import Document

chunk = Document(
    page_content=('등으로 피부의 변색, 모발의 결손, 조직(뼈, 피<br>부 등)의 결손 및 함몰 등으로 성형수술을 하여도 더<br>이상 추상(추한 '
 '모습)이 없어지지 않는 경우를 말한<br>다.<br>4) 다발성 반흔(흉터) 발생시 각 판정부위(얼굴, 머리,<br>목) 내의 다발성 '
 '반흔(흉터)의 길이 또는 면적은 합<br>산하여 평가한다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'head', 'skin']},
 'indexing': {'chunk_id': 'chunk_000968',
              'chunk_char_len': 179,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
