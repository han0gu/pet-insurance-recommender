from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 안구가 적출되어 눈자위의 조직요몰(凹<br>沒) 등으로 의안마저 끼워 넣을 수 없는 상태이면 ‘뚜렷한 '
 "추상(추한<br>모습)’으로, 의안을 끼워 넣을 수 있는 상태이면 ‘약간의 추상(추한</p><br><p id='179' "
 "data-category='paragraph' style='font-size:14px'>모습)’으로 지급률을 가산한다.<br>12) "
 '‘눈꺼풀에 뚜렷한 결손을 남긴 때’에 해당하는 경우에는 추상(추한 모<br>공<br>습)장해를 포함하여 장해를 평가한 것으로 보고 '
 '추상(추한'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_001490',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
