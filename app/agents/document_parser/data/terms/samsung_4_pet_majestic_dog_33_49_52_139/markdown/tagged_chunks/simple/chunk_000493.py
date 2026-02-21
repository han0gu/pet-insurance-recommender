from langchain_core.documents import Document

chunk = Document(
    page_content=('서에서 질문한 사항에 대하여 알고 있는 사실을 반드시 사실대로 알려야(이하「계약 전\n'
 '알릴 의무」라 하며, 상법상 「고지의무」와 같습니다) 합니다.<관련법규>[상법 제651조(고지의무위반으로 인한 계약해지)]\n'
 '보험계약당시에 보험계약자 또는 피보험자가 고의 또는 중대한 과실로 인하여 중요한 사항을 고지\n'
 '하지 아니하거나 부실의 고지를 한 때에는 보험자는 그 사실을 안 날부터 1월내에, 계약을 체결한\n'
 '날부터 3년내에 한하여 계약을 해지할 수 있다. 그러나 보험자가 계약당시에 그 사실을 알았거나'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000493',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
