from langchain_core.documents import Document

chunk = Document(
    page_content=('- 장해에 대해서는 인정하지 않는다.\n'
 '- 다) ‘정신행동에 극심한 장해를 남긴 때’라 함은 장해판정 직전 1년 이\n'
 '- 상 지속적인 정신건강의학과의 치료를 받았으며 GAF 30점 이하인 상\n'
 '- 태를 말한다.\n'
 '- 라) ‘정신행동에 심한 장해를 남긴 때’라 함은 장해판정 직전 1년 이상\n'
 '- 지속적인 정신건강의학과의 치료를 받았으며 GAF 40점 이하인 상태\n'
 '- 를 말한다.\n'
 '- 마) ‘정신행동에 뚜렷한 장해를 남긴 때’라 함은 장해판정 직전 1년 이\n'
 '- 상 지속적인 정신건강의학과의 치료를 받았으며, 보건복지부고시'),
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
 'indexing': {'chunk_id': 'chunk_000935',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
