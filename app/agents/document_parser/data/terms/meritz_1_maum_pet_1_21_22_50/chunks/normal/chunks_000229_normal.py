from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 피보험자의 가족관계등록부상 또는 주민등록상의 배우자 2. 피보험자의 3촌 이내의 친족\n'
 '② 제1항에도 불구하고, 지정대리청구인이 지정된 이후에 제1조(적용대상)의 보험수익자가 변경되는 경우에는 이미 지정된 지정대리청구인의 '
 '자격은 자동적으로 상실된 것으로 봅니다.\n'
 '제4조(지정대리청구인의 변경지정)\n'
 '① 계약자는 다음의 서류를 제출하고 지정대리청구인을 변경 지정할 수 있습니다. 이 경우 회사는 변경 지정을 서면으로 알리거나 보험증권의 '
 '뒷면에 기재하여 드립니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 42},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000229',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
