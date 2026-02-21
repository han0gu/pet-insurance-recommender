from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 한 눈의 안구(눈동자)의 주시야(머리를 움직이지 않고 눈만을 움직\n'
 '여서 볼 수 있는 범위)의 운동범위가 정상의 1/2 이하로 감소된 경우\n'
 '나) 중심 20도 이내에서 복시(물체가 둘로 보이거나 겹쳐 보임)를 남긴\n'
 '경우- \n'
 '- 7) ‘안구(눈동자)의 뚜렷한 조절기능장해’라 함은 조절력이 정상의 1/2\n'
 '- 이하로 감소된 경우를 말한다. 다만, 조절력의 감소를 무시할 수 있는\n'
 '- 50세 이상(장해진단시 연령 기준)의 경우에는 제외한다.\n'
 '- 8) ‘뚜렷한 시야 장해’라 함은 한 눈의 시야 범위가 정상시야 범위의 60%'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'eye', 'head']},
 'indexing': {'chunk_id': 'chunk_000842',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
