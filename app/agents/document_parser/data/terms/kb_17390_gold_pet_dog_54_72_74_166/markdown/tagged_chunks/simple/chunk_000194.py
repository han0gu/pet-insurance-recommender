from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 피보험자의 3촌 이내의 친족\n'
 '\uf000 제1항에도 불구하고, 지정대리청구인이 지정된 이후에 제38조(적용대상)의 보험수익자가 변경되는 경우에는 이미 지정된 '
 '지정대리청구인의 자격은 자동적으로 상실\n'
 '된 것으로 봅니다. 법\n'
 'ㆍ제40조(지정대리청구인의 변경지정)계약자는 계약체결 이후 다음의 서류를습니다. 이 경우 회사는 변경 지정을 서면으로 알리거나 보험증권의 '
 '뒷면에 기재하여\n'
 '드립니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000194',
              'chunk_char_len': 214,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
