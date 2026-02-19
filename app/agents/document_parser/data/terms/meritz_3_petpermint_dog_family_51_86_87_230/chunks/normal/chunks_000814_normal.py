from langchain_core.documents import Document

chunk = Document(
    page_content=('나) 정신행동장해는 장해판정 직전 1년 이상 충분한 정 신건강의학과의 전문적 치료를 받은 후 치료에도 불구하고 장해가 고착되었을 때 '
 '판정하여야 하며, 그렇지 않은 경우에는 그로써 고정되거나 중하게 된 장해에 대해서는 인정하지 않는다. 다) “정신행동에 극심한 장해를 '
 '남긴 때”라 함은 장 해판정 직전 1년 이상 지속적인 정신건강의학과의 치료를 받았으며 GAF 30점 이하인 상태를 말한다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 227},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000814',
              'chunk_char_len': 216,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
