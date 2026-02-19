from langchain_core.documents import Document

chunk = Document(
    page_content=('. ⑤ 제1항 제2호에 의한 특별약관의 해지가 보험금 지급사유 발생 및 보험료 납입면제사 유 발생 후에 이루어진 경우에는 '
 '제16조(상해보험계약 후 알릴 의무) 제4항 또는 제5 항에 따라 보험금을 지급하거나, 보험료 납입면제를 적용합니다. ⑥ 제1항에도 '
 '불구하고 알릴 의무를 위반한 사실이 보험금 지급사유 발생 및 보험료 납 입면제사유 발생에 영향을 미쳤음을 회사가 증명하지 못한 경우에는 '
 '제4항 및 제5항 에 관계없이 약정한 보험금을 지급하거나, 보험료 납입면제를 적용합니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 50},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000206',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
