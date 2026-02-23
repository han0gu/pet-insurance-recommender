from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 때 단체보험의 보험수익자를 피보험자 또<br>는 그 상속인이 아닌 자로 지정할 때에는 단체의 규약에서 명시적으로 정한 '
 '경<br>우가 아니면 이를 적용합니다.<br>2. 만15세 미만자, 심신상실자(心神喪失者) 또는 심신박약자(心神薄弱者)를 '
 '피보<br>험자로 하여 사망을 보험금 지급사유로 한 계약의 경우. 다만, 심신박약자가<br>계약을 체결하거나 소속 단체의 규약에 따라 '
 '단체보험의 피보험자가 될 때에<br>의사능력이 있는 경우에는 계약이 유효합니다.<br>3'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000188',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
