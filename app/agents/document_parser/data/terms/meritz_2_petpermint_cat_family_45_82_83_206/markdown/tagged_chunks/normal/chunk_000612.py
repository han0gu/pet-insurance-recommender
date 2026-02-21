from langchain_core.documents import Document

chunk = Document(
    page_content=('- 불구하고 장해가 고착되었을 때 판정하여야 하며,\n'
 '- 그렇지 않은 경우에는 그로써 고정되거나 중하게\n'
 '- 된 장해에 대해서는 인정하지 않는다.\n'
 '- 다) “정신행동에 극심한 장해를 남긴 때”라 함은 장\n'
 '- 해판정 직전 1년 이상 지속적인 정신건강의학과의\n'
 '- 치료를 받았으며 GAF 30점 이하인 상태를 말한다.\n'
 '- 라) “정신행동에 심한 장해를 남긴 때”라 함은 장해\n'
 '- 판정 직전 1년 이상 지속적인 정신건강의학과의 치\n'
 '- 료를 받았으며 GAF 40점 이하인 상태를 말한다.\n'
 '- 마) “정신행동에 뚜렷한 장해를 남긴 때”라 함은 장'),
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
 'indexing': {'chunk_id': 'chunk_000612',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
