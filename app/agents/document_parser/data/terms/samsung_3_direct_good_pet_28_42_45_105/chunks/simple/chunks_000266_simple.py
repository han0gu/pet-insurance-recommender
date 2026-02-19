from langchain_core.documents import Document

chunk = Document(
    page_content=('제31조 (강제집행 등으로 인하여 해지된 특별약관의 특별부활(효력회복))\n'
 '① 회사는 계약자의 해약환급금 청구권에 대한 강제집행, 담보권실행, 국세 및 지방세 체\n'
 '납처분절차에 따라 계약이 해지된 경우 해지 당시의 보험수익자가 계약자의 동의를\n'
 '얻어 계약 해지로 회사가 채권자에게 지급한 금액을 회사에 지급하고 제23조(특별약'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 55},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000266',
              'chunk_char_len': 179,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
