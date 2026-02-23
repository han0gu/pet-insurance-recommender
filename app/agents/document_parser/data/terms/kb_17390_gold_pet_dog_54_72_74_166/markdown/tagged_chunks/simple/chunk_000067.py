from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항의 통지에 따라 위험의 증가로 보험료를 더 내야 할 경우 회사가 청구한 추가\n'
 '보험료(정산금액을 포함합니다)를 계약자가 납입하지 않았을 때, 회사는 위험이\n'
 '증가되기 전에 적용된 보험요율(이하 "변경전 요율"이라 합니다)의 위험이 증가된\n'
 '후에 적용해야 할 보험요율(이하 "변경후 요율"이라 합니다)에 대한 비율에 따라\n'
 '보험금을 삭감하여 지급합니다. 다만, 증가된 위험과 관계없이 발생한 보험금 지\n'
 '급사유에 관해서는 원래대로 지급합니다.-'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000067',
              'chunk_char_len': 247,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
