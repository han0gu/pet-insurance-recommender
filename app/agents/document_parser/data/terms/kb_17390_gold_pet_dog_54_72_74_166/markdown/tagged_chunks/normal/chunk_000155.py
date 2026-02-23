from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에 따른 보험료의 납입최고(독촉)기간이 지나기 전까지 회사가 정한 방법에 따라\n'
 '- 보험료의 자동대출납입을 신청할 수 있으며, 이 경우 제35조(보험계약대출) 제1항\n'
 '- 에 따른 보험계약대출금으로 보험료가 자동으로 납입되어 계약은 유효하게 지속됩\n'
 '- 니다. 다만, 계약자가 서면 이외에 인터넷 또는 전화(음성녹음) 등으로 자동대출\n'
 '- 납입을 신청할 경우 회사는 자동대출납입 신청내역을 서면 또는 전화(음성녹음)\n'
 '- 등으로 계약자에게 알려드립니다.\n'
 '- \uf000 제1항의 규정에 의한 대출금과 보험료의 자동대출 납입일의 다음날부터 그 다음 보'),
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
 'indexing': {'chunk_id': 'chunk_000155',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
