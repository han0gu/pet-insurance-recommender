from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 갱신일에 있어서 피보험자의 연령 또는 피보험자의 반려동물 연령이 회사가 정\n'
 '- 한 연령의 범위 내일 것\n'
 '- 3. 갱신전 보장특약의 보험료가 정상적으로 납입완료 되었을 것\n'
 '- 제1항에 따라 자동 갱신되는 경우 보험계약 청약서에 기재된 사항 및 보험증권에\n'
 '# \uf000회사가 승인한 사항에 대하여 변경이 생긴 경우에는 계약자 또는 피보험자가 서면\n'
 '으로 그 사실을 회사에 알리고 보험증권에 확인을 받아야 합니다.\n'
 '\uf000 알릴의무에 대하여는 보통약관 제1절 일반조항 제15조(상해보험계약 후 알릴 의\n'
 '무)를 적용합니다.-'),
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
 'indexing': {'chunk_id': 'chunk_000776',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
