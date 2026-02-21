from langchain_core.documents import Document

chunk = Document(
    page_content=('분납이 제한될 수 있습니다.\n'
 '④ 제1항의 통지에 따라 위험의 증가로 보험료를 더 내야 할 경우 회사가 청구한 추가보\n'
 '험료(정산금액을 포함합니다)를 계약자가 납입하지 않았을 때, 회사는 위험이 증가되기\n'
 '전에 적용된 보험요율(이하 ‘변경전 요율’이라 합니다)의 위험이 증가된 후에 적용해야\n'
 '할 보험요율(이하 ‘변경후 요율’이라 합니다)에 대한 비율에 따라 보험금을 삭감하여 지\n'
 '급합니다. 다만, 증가된 위험과 관계없이 발생한 보험금 지급사유에 관해서는 원래대로\n'
 '지급합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000052',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
