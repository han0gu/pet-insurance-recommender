from langchain_core.documents import Document

chunk = Document(
    page_content=('① 이 약관에 따른 해약환급금은 "보험료 및 해약환급금 산출방법서" 에 따라 계산합니 다. ② 해약환급금의 지급사유가 발생한 경우 '
 '계약자는 회사에 해약환급금을 청구하여야 하 며, 회사는 청구를 접수한 날부터 3영업일 이내에 해약환급금을 지급합니다. 해약환급 금 '
 '지급일까지의 기간에 대한 이자의 계산은 보험금을 지급할 때의 적립이율 계산([별 표1] 참조)에 따릅니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 40},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000131',
              'chunk_char_len': 204,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
