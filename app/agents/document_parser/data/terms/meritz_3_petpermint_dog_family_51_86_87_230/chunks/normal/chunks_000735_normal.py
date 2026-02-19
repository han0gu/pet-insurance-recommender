from langchain_core.documents import Document

chunk = Document(
    page_content=('1) “외모”란 얼굴(눈, 코, 귀, 입 포함), 머리, 목을 말한다. 2) “추상(추한 모습)장해”라 함은 성형수술(반흔(흉 '
 '터)성형술, 레이저치료 등 포함)을 시행한 후에도 영구히 남게 되는 상태의 추상(추한 모습)을 말한다. 3) “추상(추한 모습)을 남긴 '
 '때”라 함은 상처의 흔적, 화상 등으로 피부의 변색, 모발의 결손, 조직(뼈, 피 부 등)의 결손 및 함몰 등으로 성형수술을 하여도 더 '
 '이상 추상(추한 모습)이 없어지지 않는 경우를 말한 다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 209},
 'term_type': 'special',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['skin', 'eye', 'head']},
 'indexing': {'chunk_id': 'chunk_000735',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
