from langchain_core.documents import Document

chunk = Document(
    page_content=('- 가된 수치로 평가한다.\n'
 '- 12) "가관절주)이 남아 뚜렷한 장해를 남긴 때"라 함은\n'
 '- 대퇴골에 가관절이 남은 경우 또는 경골과 종아리\n'
 '- 뼈의 2개뼈 모두에 가관절이 남은 경우를 말한다.\n'
 '※ 주) 가관절이란, 충분한 경과 및 골이식술 등 골\n'
 '유합을 얻는데 필요한 수술적 치료를 시행하\n'
 '였음에도 불구하고 골절부의 유합이 이루어지\n'
 '지 않는 ‘불유합’ 상태를 말하며, 골유합이\n'
 '지연되는 지연유합은 제외한다.- 13) "가관절이 남아 약간의 장해를 남긴 때"라 함은 경\n'
 '- 골과 종아리뼈중 어느 한 뼈에 가관절이 남은 경우'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000660',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
