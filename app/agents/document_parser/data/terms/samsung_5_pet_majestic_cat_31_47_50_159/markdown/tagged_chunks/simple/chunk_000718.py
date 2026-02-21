from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 제1항에도 불구하고 지정대리청구인이 지정된 이후에 제1조(적용대상)의 보험수익자\n'
 '- 가 변경되는 경우에는 이미 지정된 지정대리청구인의 자격은 자동적으로 상실된 것으\n'
 '- 로 봅니다.\n'
 '- ③ 제1항에도 불구하고 보험계약에서 지정대리청구인의 지정 기간을 별도로 제한한 경\n'
 '- 우, 계약자는 이 특별약관에서도 그 기간에 한하여 지정대리청구인을 지정할 수 있습\n'
 '- 니다.\n'
 '# 제4조 (지정대리청구인의 변경지정)① 계약자는 이 특별약관의 계약체결 이후 다음의 서류를 제출하고 지정대리청구인을 변'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000718',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
