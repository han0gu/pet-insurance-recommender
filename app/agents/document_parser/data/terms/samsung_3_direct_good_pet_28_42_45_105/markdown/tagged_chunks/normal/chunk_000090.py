from langchain_core.documents import Document

chunk = Document(
    page_content=('든 채무액을 뺀 금액을 초과하는 경우에는 보험료의 자동대출납입을 더는 할 수 없습\n'
 '니다.# <용어풀이># [보험계약대출이율]해당 보험상품의 약관에 따라 계약자가 대출을 받을 경우, 회사가 정하는 대출이율이며, 이 '
 '계약의\n'
 '보험계약대출이율이 변경되는 경우, 변경된 시점부터 변경된 이율을 적용합니다.# 따라 재신청을 하여야 합니다.④ 보험료의 자동대출납입이 '
 '행하여진 경우에도 자동대출납입 전 납입최고(독촉)기간이\n'
 '끝나는 날의 다음날부터 1개월 이내에 계약자가 계약의 해지를 청구한 때에는 회사는'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000090',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
