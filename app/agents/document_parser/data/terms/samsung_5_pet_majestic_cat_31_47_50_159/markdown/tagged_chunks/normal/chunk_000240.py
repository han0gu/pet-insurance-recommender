from langchain_core.documents import Document

chunk = Document(
    page_content=('료의 자동대출납입을 신청할 수 있으며, 이 경우 제36조(보험계약대출) 제1항에 따른\n'
 '보험계약대출금으로 보험료가 자동으로 납입되어 계약은 유효하게 지속됩니다. 다만,\n'
 '계약자가 서면 이외에 인터넷 또는 전화(음성녹음) 등으로 자동대출납입을 신청할 경\n'
 '우 회사는 자동대출납입 신청내역을 서면, 전화(음성녹음) 또는 전자문서(SMS포함)\n'
 '등으로 계약자에게 알려 드립니다.<용어풀이># [자동대출납입]보험료를 제때에 납입하기 곤란한 경우에 계약자가 자동대출납입을 신청하면 해당 '
 '보험 상품의'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000240',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
