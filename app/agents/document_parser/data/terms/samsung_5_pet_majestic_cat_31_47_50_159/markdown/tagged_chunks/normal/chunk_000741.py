from langchain_core.documents import Document

chunk = Document(
    page_content=('- 상 덮거나 또는 눈을 감았을 때 각막을 완전히 덮을 수 없는 경우를 말한다.\n'
 '- 11) 외상이나 화상 등으로 안구의 적출이 불가피한 경우에는 외모의 추상(추한 모\n'
 '- 습)이 가산된다. 이 경우 안구가 적출되어 눈자위의 조직요몰(凹沒) 등으로 의\n'
 '- 안마저 끼워 넣을 수 없는 상태이면 "뚜렷한 추상(추한 모습)" 으로, 의안을\n'
 '- 끼워 넣을 수 있는 상태이면 "약간의 추상(추한 모습)" 으로 지급률을 가산한\n'
 '- 다.\n'
 '12) "눈꺼풀에 뚜렷한 결손을 남긴 때" 에 해당하는 경우에는 추상(추한 모습)장'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000741',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
