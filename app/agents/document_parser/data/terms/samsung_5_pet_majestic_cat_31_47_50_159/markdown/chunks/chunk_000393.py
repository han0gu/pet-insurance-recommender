from langchain_core.documents import Document

chunk = Document(
    page_content=('정하는 바에 따라 회사가 적립한 사망당시 이 특별약관의 계약자적립액 및 미경과보험료\n'
 '를 계약자에게 지급하고, 이 특별약관은 더 이상 효력이 없습니다.- 85 -1-12. 반려동물 양육자금Ⅰ 특별약관# 제1관 일반사항- ① '
 '제2관 개별사항에서 정하지 않은 사항은 특별약관의 일반사항을 적용합니다.\n'
 '- ② 제1항에도 불구하고, 이 상품의 사업방법서 별지에 따라 납입면제를 적용하지 않거나\n'
 '- 보통약관에서 보험료 납입면제에 대해 정하지 않은 경우 특별약관 일반사항 제5조(보'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
