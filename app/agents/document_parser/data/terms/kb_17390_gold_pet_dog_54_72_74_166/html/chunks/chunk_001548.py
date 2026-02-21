from langchain_core.documents import Document

chunk = Document(
    page_content=(". 이하 이 신체부위에서 같다)의 압박률 또는 척추체(척추<br>뼈 몸통)의 만곡 정도에 따라 평가한다.</p><br><p id='38' "
 "data-category='list' style='font-size:16px'>가) 척추체(척추뼈 몸통)의 만곡변화는 객관적인 "
 "측정방법(Cobb's<br>Angle)에 따라 골절이 발생한 척추체(척추뼈 몸통)의 상․하 인접 정<br>상 척추체(척추뼈 몸통)를 "
 '포함하여 측정하며, 생리적 정상만곡을<br>고려하여 평가한다.<br>나) 척추(등뼈)의 기형장해는 척추체(척추뼈 몸통)의 압박률,'),
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
