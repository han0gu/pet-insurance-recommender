from langchain_core.documents import Document

chunk = Document(
    page_content=("척추(등뼈)의 골절 또는 탈구 등으로 35° 이상의 척추전만증(척추</p><br><p id='49' "
 "data-category='paragraph' style='font-size:16px'>가 앞으로 휘어지는 증상), 척추후만증(척추가 "
 '뒤로 휘어지는 증상)<br>또는 20° 이상의 척추측만증(척추가 옆으로 휘어지는 증상) 변형이<br>있을 때<br>나) 척추체(척추뼈 '
 '몸통) 한 개의 압박률이 60%이상인 경우 또는 한 운<br>동단위 내에 두 개 이상 척추체(척추뼈 몸통)의 압박골절로 각 '
 '척추<br>체(척추뼈 몸통)의 압박률의 합이'),
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
