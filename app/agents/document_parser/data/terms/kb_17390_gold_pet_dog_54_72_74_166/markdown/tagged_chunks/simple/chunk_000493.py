from langchain_core.documents import Document

chunk = Document(
    page_content=('- 제1항에 의한 계약의 해지가 손해발생 전에 이루어진 경우에는 보통약관 제1절\n'
 '- 일반조항 제34조(해약환급금) 제1항에 의한 해약환급금을 계약자에게 지급합\n'
 '- 니다.\n'
 '- 제1항 제1호에 의한 계약의 해지가 보험금 지급사유 발생 후에 이루어진 경우에\n'
 '- 104 -회사는 보험금을 지급하지 않으며, 계약 전 알릴 의무 위반사실(계약해지 등의\n'
 '원인이 되는 위반사실을 구체적으로 명시)뿐만 아니라 계약 전 알릴 의무사항이\n'
 '중요한 사항에 해당되는 사유를 "반대증거가 있는 경우 이의를 제기할 수 있습니'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000493',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
