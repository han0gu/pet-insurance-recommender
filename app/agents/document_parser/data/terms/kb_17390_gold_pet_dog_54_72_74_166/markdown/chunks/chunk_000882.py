from langchain_core.documents import Document

chunk = Document(
    page_content=('또는 20° 이상의 척추측만증(척추가 옆으로 휘어지는 증상) 변형이\n'
 '있을 때\n'
 '나) 척추체(척추뼈 몸통) 한 개의 압박률이 60%이상인 경우 또는 한 운\n'
 '동단위 내에 두 개 이상 척추체(척추뼈 몸통)의 압박골절로 각 척추\n'
 '체(척추뼈 몸통)의 압박률의 합이 90% 이상일 때\n'
 '뚜렷한 기형이란 다음 중 어느 하나에 해당하는 경우를 말한다.- \n'
 '10)KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 145- 145 -법ㆍ규정가) 척추(등뼈)의 골절 또는 탈구 등으로 15° '
 '이상의 척추전만증(척추'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
