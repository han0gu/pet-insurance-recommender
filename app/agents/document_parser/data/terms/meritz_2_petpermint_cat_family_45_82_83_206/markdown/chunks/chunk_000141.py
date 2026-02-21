from langchain_core.documents import Document

chunk = Document(
    page_content=('니다. 이와 별도로 본 보험회사 보호상품의 사고보험금\n'
 '을 합산한 금액이 1인당 “1억원까지” 보호됩니다. 다\n'
 '만, 계약자 및 보험료납부자가 법인인 보험계약의 경우\n'
 '에는 보호되지 않습니다.82무배당 펫퍼민트 Cat&Family보험\n'
 '다이렉트2601 특별약관8384# Ⅰ. 반려동물 비용손해 관련 특별약관반려동물 비용손해 관련 특별약관 일반조항# 제1조(목적)이 '
 '특별약관은 계약자와 회사 사이에 피보험자 소유의 보험\n'
 '증권에 기재된 반려동물의 질병 또는 상해로 인한 손해를'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
