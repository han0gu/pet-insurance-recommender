from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 계약<br>자가 재가입을 원하지 않는 경우에는 해당 시점으로부터 계약은 해지됩니다(단,<br>최초연장된 날로부터 90일 '
 '이전에는 계약을 취소 또는 해지할 수 있습니다.)<br>\uf000 제7항 내지 제9항에 따라 계약이 해지된 경우 회사는 보통약관 제1절 '
 "일반조항<br>\uf000</p><br><p id='106' data-category='list'></p><br><p id='107' "
 "data-category='paragraph' style='font-size:18px'>- 108 -</p><p id='108'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000929',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
