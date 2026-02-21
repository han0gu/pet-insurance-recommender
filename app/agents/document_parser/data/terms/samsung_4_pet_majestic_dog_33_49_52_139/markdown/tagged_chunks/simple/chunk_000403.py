from langchain_core.documents import Document

chunk = Document(
    page_content=('약관 제36조(해약환급금)을 적용합니다.- \n'
 '# 제2관 개별사항# 제1조 (보험금의 지급사유)회사는 피보험자가 보험증권에 기재된 이 특별약관의 보험기간(이하「보험기간」이라 합\n'
 '니다) 중에 「특정법정감염병」으로 감염병의 예방 및 관리에 관한 법률 제11조(의사 등\n'
 '의 신고)에 따라 신고되어 특정법정감염병 환자로 진단 확정되었을 때에는 보험증권에 기\n'
 '재된 이 특별약관의 보험가입금액을 특정법정감염병 진단비로 보험수익자에게 지급합니'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000403',
              'chunk_char_len': 239,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
