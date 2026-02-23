from langchain_core.documents import Document

chunk = Document(
    page_content=('- 15) 어린이의 유치는 향후에 영구치로 대체되므로 후유장해의 대상이 되지 않으\n'
 '- 나, 선천적으로 영구치 결손이 있는 경우에는 유치의 결손을 후유장해로 평가\n'
 '- 한다.\n'
 '- 16) 가철성 보철물(신체의 일부에 붙였다 떼었다 할 수 있는 틀니 등)의 파손은 후\n'
 '- 유장해의 대상이 되지 않는다.\n'
 '- 5. 외모의 추상(추한 모습)장해\n'
 '# 가. 장해의 분류장 해 의 분 류 지급률(%)\n'
 '1) 외모에 뚜렷한 추상(추한 모습)을 남긴 때 15'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000762',
              'chunk_char_len': 243,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
