from langchain_core.documents import Document

chunk = Document(
    page_content=('- 제18조(강제집행 등으로 인하여 해지된 특별약관의 특별부활(효력회복))\n'
 '- \uf000 회사는 계약자의 해약환급금 청구권에 대한 강제집행, 담보권실행, 국세 및 지방\n'
 '- 특\n'
 '- 세 체납처분절차에 따라 특별약관이 해지된 경우 해지 당시의 보험수익자가 계약\n'
 '- 별\n'
 '- 자의 동의를 얻어 특별약관 해지로 회사가 채권자에게 지급한 금액을 회사에 지\n'
 '- 약\n'
 '- 급하고 제13조(특별약관 내용의 변경 등) 제1항의 절차에 따라 계약자 명의를 보\n'
 '- 관\n'
 '- 험수익자로 변경하여 특별약관의 특별부활(효력회복)을 청약할 수 있음을 보험수'),
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
 'indexing': {'chunk_id': 'chunk_000523',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
