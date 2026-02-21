from langchain_core.documents import Document

chunk = Document(
    page_content=('- 더 이상 효력이 없습니다.\n'
 '- 62 -- \n'
 '62 / 1812. 질병 관련 특별약관2-1. 반려동물 양육자금Ⅱ 특별약관# 제1관 일반사항- ① 제2관 개별사항에서 정하지 않은 사항은 '
 '특별약관의 일반사항을 적용합니다. 단, 특별\n'
 '- 약관 일반사항 제7조(보험금을 지급하지 않는 사유)는 제외합니다.\n'
 '- ② 제1항에도 불구하고, 이 상품의 사업방법서 별지에 따라 납입면제를 적용하지 않거나\n'
 '- 보통약관에서 보험료 납입면제에 대해 정하지 않은 경우 특별약관 일반사항 제5조(보'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
